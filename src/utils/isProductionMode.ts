import { ExtensionMode, ExtensionContext } from 'vscode';

export function isProductionMode(context: ExtensionContext): boolean {
  return context.extensionMode === ExtensionMode.Production;
}

export function isE2EDevMode(context: ExtensionContext): boolean {
  return Boolean(process.env.E2E_DEV);
}
